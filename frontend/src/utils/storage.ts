/**
 * Utility functions for local storage operations
 */

/**
 * Save data to local storage
 * @param key Storage key
 * @param value Data to store
 */
export const saveToStorage = <T>(key: string, value: T): void => {
  try {
    const serializedValue = JSON.stringify(value);
    localStorage.setItem(key, serializedValue);
  } catch (error) {
    console.error(`Error saving to localStorage: ${error}`);
  }
};

/**
 * Retrieve data from local storage
 * @param key Storage key
 * @param defaultValue Default value if key is not found
 * @returns The stored value or default value
 */
export const getFromStorage = <T>(key: string, defaultValue: T): T => {
  try {
    const serializedValue = localStorage.getItem(key);
    if (serializedValue === null) {
      return defaultValue;
    }
    return JSON.parse(serializedValue) as T;
  } catch (error) {
    console.error(`Error getting from localStorage: ${error}`);
    return defaultValue;
  }
};

/**
 * Remove data from local storage
 * @param key Storage key to remove
 */
export const removeFromStorage = (key: string): void => {
  try {
    localStorage.removeItem(key);
  } catch (error) {
    console.error(`Error removing from localStorage: ${error}`);
  }
};